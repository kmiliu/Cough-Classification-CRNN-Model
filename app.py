"""Browse verified historical results; no checkpoint or participant data is loaded."""
from pathlib import Path
import streamlit as st
from scripts.summarize_results import summarize

BASE = Path(__file__).resolve().parent


def main():
    st.set_page_config(page_title='Respiratory Sound Classification', layout='centered')
    st.title('Respiratory Sound Classification')
    st.write('Explore the recorded Keras CNN–BiGRU experiment using its saved predictions.')
    st.warning('Historical exploratory results. The decision threshold was selected on the same test set. This is not a diagnostic tool or an independent clinical evaluation.')
    try:
        results = summarize()
    except (OSError, ValueError) as exc:
        st.error(f'Result artifact unavailable or invalid: {type(exc).__name__}. See README for setup.')
        return
    left, right = st.columns(2)
    left.metric('ROC-AUC', f"{results['roc_auc']:.4f}")
    right.metric('PR-AUC (trapezoidal)', f"{results['pr_auc_trapezoidal']:.4f}")
    st.caption(f"{results['rows']:,} prediction rows; {results['positive_rows']} positive and {results['negative_rows']:,} negative labels. Rows are not established as independent participants.")
    st.subheader('Recorded thresholded predictions')
    st.write({key: round(results[key], 4) for key in ['positive_precision', 'positive_recall', 'positive_f1']})
    st.table({'Actual class': ['0', '1'], 'Predicted 0': [results['confusion_matrix_actual_by_predicted'][0][0], results['confusion_matrix_actual_by_predicted'][1][0]], 'Predicted 1': [results['confusion_matrix_actual_by_predicted'][0][1], results['confusion_matrix_actual_by_predicted'][1][1]]})
    st.subheader('Existing evaluation curves')
    folder = BASE / 'output/crnn_model_results_5'
    for name, caption in [('roc_curve_crnn_focal_biGRU_v2.png', 'Historical ROC curve'), ('pr_curve_crnn_focal_biGRU_v2.png', 'Historical precision–recall curve')]:
        if (folder / name).is_file():
            st.image(str(folder / name), caption=caption)
    st.info('Live audio prediction is unavailable: the recorded experiment used 49 numeric inputs, while the old audio demo supplied 29. A matching Keras checkpoint, feature schema, and fitted preprocessing artifacts are required before inference can be restored.')
    st.caption('Metrics are recomputed from the committed prediction CSV. No audio, participant metadata, or model upload is requested.')


if __name__ == '__main__':
    main()
