from __future__ import annotations

import os
from glob import glob
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"
MPL_DIR = CACHE_DIR / "matplotlib"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image, UnidentifiedImageError
import tensorflow as tf


def find_project_root(start_path: Path) -> Path:
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Could not locate the project data directory.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "skin_cancer_cnn.keras"
STATS_PATH = ARTIFACTS_DIR / "preprocessing_stats.npz"

LESION_TYPE_DICT = {
    "nv": "Melanocytic nevi",
    "mel": "Melanoma",
    "bkl": "Benign keratosis-like lesions",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratoses",
    "vasc": "Vascular lesions",
    "df": "Dermatofibroma",
}
CLASS_NAMES = list(LESION_TYPE_DICT.values())
LABEL_LOOKUP = {diagnosis_code: idx for idx, diagnosis_code in enumerate(LESION_TYPE_DICT.keys())}


st.set_page_config(page_title="HAM10000 Dashboard", layout="wide")
sns.set_theme(style="whitegrid")

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }
    .hero {
        padding: 1.8rem 2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #0f2747 0%, #1f5c7a 45%, #8fb9b0 100%);
        color: #f7fbff;
        margin-bottom: 1rem;
        box-shadow: 0 20px 45px rgba(15, 39, 71, 0.18);
    }
    .hero h1 {
        margin: 0;
        font-size: 2.5rem;
        line-height: 1.05;
    }
    .hero p {
        margin: 0.7rem 0 0 0;
        max-width: 760px;
        font-size: 1rem;
        line-height: 1.6;
    }
    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 1rem;
    }
    .pill {
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.18);
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_metadata() -> pd.DataFrame:
    imageid_path_dict = {
        Path(image_path).stem: image_path
        for image_path in glob(str(DATA_DIR / "HAM10000_images_part_*" / "*.jpg"))
    }

    metadata = pd.read_csv(DATA_DIR / "HAM10000_metadata.csv")
    metadata["path"] = metadata["image_id"].map(imageid_path_dict)
    metadata = metadata.dropna(subset=["path"]).copy()
    metadata["cell_type"] = metadata["dx"].map(LESION_TYPE_DICT)
    metadata["cell_type_idx"] = metadata["dx"].map(LABEL_LOOKUP)
    metadata["age"] = metadata["age"].fillna(metadata["age"].mean())
    metadata["sex"] = metadata["sex"].fillna("unknown")
    metadata["localization"] = metadata["localization"].fillna("unknown")
    return metadata


@st.cache_data(show_spinner=False)
def load_preprocessing_stats() -> dict | None:
    if not STATS_PATH.exists():
        return None

    with np.load(STATS_PATH, allow_pickle=True) as stats:
        return {
            "train_mean": float(stats["train_mean"]),
            "train_std": float(stats["train_std"]),
            "class_names": [str(name) for name in stats["class_names"].tolist()],
            "image_width": int(stats["image_width"]),
            "image_height": int(stats["image_height"]),
        }


@st.cache_resource(show_spinner=False)
def load_trained_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH)


def build_count_figure(
    data: pd.DataFrame,
    column: str,
    title: str,
    color: str,
    order: list[str] | None = None,
    horizontal: bool = False,
    limit: int | None = None,
):
    counts = data[column].value_counts()
    if order is not None:
        counts = counts.reindex([item for item in order if item in counts.index]).dropna()
    if limit is not None:
        counts = counts.head(limit)

    fig, ax = plt.subplots(figsize=(8, 5))
    if horizontal:
        sns.barplot(x=counts.values, y=counts.index, ax=ax, color=color)
        ax.set_xlabel("Images")
        ax.set_ylabel("")
    else:
        sns.barplot(x=counts.index, y=counts.values, ax=ax, color=color)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def build_age_figure(data: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=data, x="age", bins=30, ax=ax, color="#c65d3b")
    ax.set_title("Age Distribution")
    ax.set_xlabel("Age")
    fig.tight_layout()
    return fig


def preprocess_image_for_model(image: Image.Image, stats: dict):
    resized_image = image.convert("RGB").resize((stats["image_width"], stats["image_height"]))
    image_array = np.asarray(resized_image, dtype=np.float32)
    normalized = (image_array - stats["train_mean"]) / max(stats["train_std"], 1e-7)
    return np.expand_dims(normalized, axis=0), resized_image


def render_prediction_panel(source_image: Image.Image, stats: dict, model) -> None:
    model_input, preview_image = preprocess_image_for_model(source_image, stats)
    probabilities = model.predict(model_input, verbose=0)[0]
    class_names = stats["class_names"]
    if len(class_names) != len(probabilities):
        class_names = CLASS_NAMES[: len(probabilities)]

    prediction_df = pd.DataFrame(
        {
            "Lesion type": class_names,
            "Probability": probabilities,
        }
    ).sort_values("Probability", ascending=False)

    top_prediction = prediction_df.iloc[0]
    preview_col, result_col = st.columns([1, 1.2])
    with preview_col:
        st.image(preview_image, caption="Model input after resize", width="stretch")
    with result_col:
        st.metric("Top prediction", top_prediction["Lesion type"], f"{top_prediction['Probability']:.1%}")
        st.bar_chart(prediction_df.set_index("Lesion type"))
        st.dataframe(
            prediction_df.assign(Probability=prediction_df["Probability"].map(lambda value: f"{value:.2%}")),
            width="stretch",
            hide_index=True,
        )


metadata_df = load_metadata()

st.markdown(
    """
    <div class="hero">
        <h1>HAM10000 Skin Cancer Dashboard</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Filters")
selected_lesions = st.sidebar.multiselect("Lesion type", CLASS_NAMES, default=CLASS_NAMES)
sex_options = sorted(metadata_df["sex"].unique().tolist())
selected_sex = st.sidebar.multiselect("Sex", sex_options, default=sex_options)
age_min = int(metadata_df["age"].min())
age_max = int(metadata_df["age"].max())
selected_age = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))

filtered_df = metadata_df[
    metadata_df["cell_type"].isin(selected_lesions)
    & metadata_df["sex"].isin(selected_sex)
    & metadata_df["age"].between(selected_age[0], selected_age[1])
].copy()

overview_tab, eda_tab, samples_tab, prediction_tab = st.tabs(
    ["Overview", "EDA", "Samples", "Prediction"]
)

with overview_tab:
    total_images = len(filtered_df)
    lesion_classes = int(filtered_df["cell_type"].nunique()) if total_images else 0
    mean_age = float(filtered_df["age"].mean()) if total_images else 0.0
    dominant_class = (
        filtered_df["cell_type"].value_counts().idxmax()
        if total_images
        else "No rows match the current filters"
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Visible images", f"{total_images:,}")
    metric_columns[1].metric("Visible classes", lesion_classes)
    metric_columns[2].metric("Average age", f"{mean_age:.1f}")
    metric_columns[3].metric("Most common lesion", dominant_class)


    st.dataframe(
        filtered_df[
            ["image_id", "cell_type", "age", "sex", "localization", "dx_type"]
        ].sort_values(["cell_type", "image_id"]),
        width="stretch",
        height=460,
        hide_index=True,
    )

with eda_tab:
    if filtered_df.empty:
        st.warning("No rows match the current filters.")
    else:
        top_left, top_right = st.columns(2)
        bottom_left, bottom_right = st.columns(2)

        with top_left:
            fig = build_count_figure(
                filtered_df,
                column="cell_type",
                title="Lesion Type Distribution",
                color="#1f5c7a",
                order=CLASS_NAMES,
                horizontal=True,
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with top_right:
            fig = build_age_figure(filtered_df)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with bottom_left:
            fig = build_count_figure(
                filtered_df,
                column="sex",
                title="Sex Distribution",
                color="#5a9367",
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with bottom_right:
            fig = build_count_figure(
                filtered_df,
                column="localization",
                title="Top Lesion Localizations",
                color="#2f4858",
                horizontal=True,
                limit=10,
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

with samples_tab:
    if filtered_df.empty:
        st.warning("No rows match the current filters.")
    else:
        available_lesions = [lesion for lesion in CLASS_NAMES if lesion in filtered_df["cell_type"].unique()]
        selected_gallery_lesion = st.selectbox("Lesion gallery", available_lesions)
        lesion_df = filtered_df[filtered_df["cell_type"] == selected_gallery_lesion]
        sample_max = max(1, min(12, len(lesion_df)))
        sample_default = min(6, sample_max)
        sample_count = st.slider("Samples to display", 1, sample_max, sample_default)
        gallery_df = lesion_df.sample(n=min(sample_count, len(lesion_df)), random_state=1234)

        gallery_columns = st.columns(3)
        for index, (_, row) in enumerate(gallery_df.iterrows()):
            with gallery_columns[index % 3]:
                st.image(row["path"], caption=f"{row['image_id']} • {row['sex']} • age {int(row['age'])}", width="stretch")
                st.caption(f"{row['localization']} • {row['dx_type']}")

from PIL import Image, UnidentifiedImageError
import streamlit as st

with prediction_tab:
    stats = load_preprocessing_stats()
    model = load_trained_model()

    st.subheader("AI Skin Cancer Detection")

    st.caption(
        "Upload a dermoscopic skin lesion image to analyze and identify the detected skin cancer category using the trained deep learning model."
    )

    missing_artifacts = []

    if stats is None:
        missing_artifacts.append(
            f"`{STATS_PATH.relative_to(PROJECT_ROOT)}`"
        )

    if model is None:
        missing_artifacts.append(
            f"`{MODEL_PATH.relative_to(PROJECT_ROOT)}`"
        )

    if missing_artifacts:
        st.warning(
            "Required trained model files are missing: "
            + ", ".join(missing_artifacts)
        )

        st.caption(
            "Run the training notebook in `src/notebook.ipynb` "
            "to generate the trained model and preprocessing statistics."
        )

    else:
        uploaded_file = st.file_uploader(
            "Upload Skin Lesion Image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG",
        )

        if uploaded_file is None:
            st.info(
                "Upload a skin lesion image to begin AI-based detection."
            )

        else:
            try:
                uploaded_image = Image.open(uploaded_file).convert("RGB")

                st.image(
                    uploaded_image,
                    caption="Uploaded Skin Lesion Image",
                    use_container_width=True,
                )

                with st.spinner("Analyzing image and detecting skin cancer type..."):
                    render_prediction_panel(
                        uploaded_image,
                        stats,
                        model,
                    )

            except UnidentifiedImageError:
                st.error(
                    "The uploaded file is not a valid image. Please upload a JPG or PNG image."
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")
