import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from docx import Document
from docx.shared import Inches
import shutil


def _load_jsons(pattern: str):
    files = Path("outputs/metrics").glob(pattern)
    data = []
    for f in files:
        try:
            data.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue
    return pd.DataFrame(data)


def _load_history(pattern: str):
    files = Path("outputs/history").glob(pattern)
    rows = []
    for f in files:
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
            dataset = f.stem.split("__")[1] if "__" in f.stem else "unknown"
            model = f.stem.split("__")[2] if "__" in f.stem and len(f.stem.split("__")) > 2 else f.stem
            for i, ep in enumerate(payload.get("epoch", [])):
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "epoch": ep,
                        "train_loss": payload.get("train_loss", [None] * len(payload.get("epoch", [])))[i],
                        "val_loss": payload.get("val_loss", [None] * len(payload.get("epoch", [])))[i],
                        "val_accuracy": payload.get("val_accuracy", [None] * len(payload.get("epoch", [])))[i],
                    }
                )
        except Exception:
            continue
    return pd.DataFrame(rows)


def _save_table(df, name):
    out_csv = Path("docs/tables") / name
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    out_xlsx = Path("docs/defense_pack") / name.replace(".csv", ".xlsx")
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_xlsx, index=False)


def _plot_bar_with_floor(series, ylabel, title, out_path, floor=1.0, note_if_zero=None):
    ax = series.plot(kind="bar")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    max_val = float(series.max()) if not series.empty else 0.0
    if max_val == 0.0:
        ax.set_ylim(0.0, floor)
        if note_if_zero:
            ax.text(
                0.5,
                0.5,
                note_if_zero,
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
    else:
        ax.set_ylim(0.0, min(1.0, max_val * 1.1))
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def make_figures(class_df, det_df):
    figs_dir = Path("docs/figures")
    figs_dir.mkdir(parents=True, exist_ok=True)

    if not class_df.empty:
        _plot_bar_with_floor(
            class_df.groupby("model")["accuracy"].mean(),
            "Accuracy",
            "Classification Accuracy Comparison",
            figs_dir / "FIG_Classification_Accuracy_Comparison.png",
        )

    if not det_df.empty:
        _plot_bar_with_floor(
            det_df.groupby("model")["map50"].mean(),
            "mAP@0.5",
            "YOLO mAP Comparison",
            figs_dir / "FIG_YOLO_mAP_Comparison.png",
            note_if_zero="All mAP@0.5 = 0.0\nCheck labels/training",
        )

        _plot_bar_with_floor(
            det_df.groupby("model")["latency_ms"].mean(),
            "Latency (ms)",
            "Latency Comparison - All Models",
            figs_dir / "FIG_Latency_Comparison_AllModels.png",
            floor=1.0,
        )

        _plot_bar_with_floor(
            det_df.groupby("model")["fps"].mean(),
            "FPS",
            "FPS Comparison - All Models",
            figs_dir / "FIG_FPS_Comparison_AllModels.png",
            floor=1.0,
        )

        _plot_bar_with_floor(
            det_df.groupby("model")["energy_proxy_wh_per_1000"].mean(),
            "Wh / 1000 inferences (proxy)",
            "Energy Proxy Comparison",
            figs_dir / "FIG_EnergyProxy_Comparison.png",
            floor=1.0,
        )

        plt.scatter(det_df["map50"], det_df["latency_ms"])
        plt.xlabel("mAP@0.5")
        plt.ylabel("Latency (ms)")
        plt.title("Tradeoff: mAP vs Latency")
        plt.tight_layout()
        plt.savefig(figs_dir / "FIG_Tradeoff_mAP_vs_Latency.png", dpi=300)
        plt.close()

        plt.scatter(det_df["map50"], det_df["energy_proxy_wh_per_1000"])
        plt.xlabel("mAP@0.5")
        plt.ylabel("Wh / 1000 inferences (proxy)")
        plt.title("Tradeoff: mAP vs Energy")
        plt.tight_layout()
        plt.savefig(figs_dir / "FIG_Tradeoff_mAP_vs_Energy.png", dpi=300)
        plt.close()


def make_sample_detections(det_df):
    figs_dir = Path("docs/figures")
    figs_dir.mkdir(parents=True, exist_ok=True)
    for _, row in det_df.drop_duplicates(["dataset", "model"]).iterrows():
        dataset = row["dataset"]
        model = row["model"]
        run_dir = Path("outputs/logs") / f"{dataset}__{model}"
        if not run_dir.exists():
            alt = Path("runs/detect/outputs/logs") / f"{dataset}__{model}"
            if alt.exists():
                run_dir = alt
        best_pt = run_dir / "weights" / "best.pt"
        if not best_pt.exists():
            continue
        sample_img = next((Path(f"data/processed/{dataset}/images_detection/test").rglob("*.jpg")), None)
        if not sample_img:
            sample_img = next((Path(f"data/processed/{dataset}/images_detection/test").rglob("*.png")), None)
        if not sample_img:
            sample_img = next((Path(f"data/processed/{dataset}/images_detection/test").rglob("*.bmp")), None)
        if not sample_img:
            continue
        y = YOLO(str(best_pt))
        res = y.predict(source=str(sample_img), save=False, verbose=False)
        if res:
            im = res[0].plot()
            out_path = figs_dir / f"FIG_YOLO_SampleDetections_{dataset}__{model}.png"
            from PIL import Image
            Image.fromarray(im).save(out_path)


def make_defense_pack():
    pack = Path("docs/defense_pack")
    pack.mkdir(parents=True, exist_ok=True)

    # copy tables/figures
    for p in Path("docs/tables").glob("*.csv"):
        shutil.copy2(p, pack / p.name)
    for p in Path("docs/figures").glob("*.png"):
        shutil.copy2(p, pack / p.name)


def make_appendix_docx():
    doc = Document()
    doc.add_heading("Results Appendix", level=1)
    doc.add_paragraph("Auto-generated defense appendix with key tables and figures.")

    for fig in Path("docs/figures").glob("*.png"):
        doc.add_heading(fig.stem, level=2)
        doc.add_picture(str(fig), width=Inches(5.5))

    out_path = Path("docs/defense_pack/Results_Appendix.docx")
    doc.save(out_path)


def main():
    class_df = _load_jsons("classification_bench__*.json")
    det_df = _load_jsons("detection_bench__*.json")
    sim_df = _load_jsons("realtime_sim__*.json")
    hist_df = _load_history("classification__*.json")

    if not class_df.empty:
        _save_table(class_df, "TABLE_Classification_Benchmark_All.csv")
    if not det_df.empty:
        _save_table(det_df, "TABLE_Detection_Benchmark_All.csv")
        _save_table(det_df[["dataset","model","map50","latency_ms","energy_proxy_wh_per_1000"]], "TABLE_YOLO_Tradeoff_Accuracy_Latency_Energy.csv")
    if not sim_df.empty:
        _save_table(sim_df, "TABLE_RealTime_Simulation.csv")
    if not hist_df.empty:
        _save_table(hist_df, "TABLE_Classification_TrainVal_History.csv")

    if not class_df.empty or not det_df.empty:
        model_sizes = []
        for df in [class_df, det_df]:
            if df.empty:
                continue
            model_sizes.append(df[["dataset","model","model_size_mb"]])
        if model_sizes:
            model_sizes_df = pd.concat(model_sizes, ignore_index=True)
            _save_table(model_sizes_df, "TABLE_Model_Size_Comparison.csv")

    make_figures(class_df, det_df)
    if not det_df.empty:
        make_sample_detections(det_df)

    # defense summary
    summary = Path("docs/defense_pack/DEFENSE_SUMMARY.md")
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        "Defense pack generated. See tables and figures in this folder.\n",
        encoding="utf-8"
    )

    # copy tables and figures into defense pack
    pack = Path("docs/defense_pack")
    pack.mkdir(parents=True, exist_ok=True)
    for p in Path("docs/tables").glob("*.csv"):
        shutil.copy2(p, pack / p.name)
    for p in Path("docs/figures").glob("*.png"):
        shutil.copy2(p, pack / p.name)

    make_appendix_docx()


if __name__ == "__main__":
    main()
