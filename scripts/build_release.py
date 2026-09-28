"""Create a release from an explicit allowlist; never include raw data or Git history."""
import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ['README.md', 'app.py', 'requirements.txt', 'requirements-training.txt',
         'scripts/summarize_results.py', 'scripts/test_model_load.py',
         'scripts/build_release.py', 'scripts/make_shareable_release.sh']
RESULT_FILES = ['verified_summary.json', 'test_predictions_f1opt_v2.csv',
                'roc_curve_crnn_focal_biGRU_v2.png', 'pr_curve_crnn_focal_biGRU_v2.png',
                'training_curves_crnn_focal_biGRU_v2.png', 'report.ipynb']


def build_release(destination, root=ROOT):
    root = Path(root).resolve()
    destination = Path(destination).resolve()
    paths = [root / name for name in FILES]
    paths += [root / 'output/crnn_model_results_5' / name for name in RESULT_FILES]
    paths += sorted((root / 'coding').glob('*.py'))
    paths += sorted((root / 'tests').glob('test_*.py'))
    for path in paths:
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f'Missing or unsafe release entry: {path.name}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting any existing file, including source.
    with zipfile.ZipFile(destination, 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(root))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', type=Path, default=Path('respiratory-results-release.zip'))
    args = parser.parse_args()
    print(f'Created {build_release(args.output).name}')


if __name__ == '__main__':
    main()
