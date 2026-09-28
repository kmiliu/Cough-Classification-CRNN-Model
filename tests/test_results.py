import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_results import summarize
from scripts.build_release import build_release


class ResultTests(unittest.TestCase):
    def score(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.csv'
            with path.open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['y_true', 'y_pred_prob', 'y_pred_label'])
                writer.writerows(rows)
            return summarize(path)

    def test_perfect_ranking(self):
        result = self.score([(0, 0.1, 0), (1, 0.9, 1)])
        self.assertEqual(result['roc_auc'], 1)
        self.assertEqual(result['pr_auc_trapezoidal'], 1)
        self.assertEqual(result['positive_f1'], 1)

    def test_tied_ranking(self):
        result = self.score([(0, 0.5, 1), (1, 0.5, 1)])
        self.assertEqual(result['roc_auc'], 0.5)
        self.assertEqual(result['pr_auc_trapezoidal'], 0.75)

    def test_reverse_ranking(self):
        self.assertEqual(self.score([(0, 0.9, 1), (1, 0.1, 0)])['roc_auc'], 0)

    def test_invalid_probabilities(self):
        with self.assertRaises(ValueError):
            self.score([(0, float('nan'), 0), (1, 0.5, 1)])
        with self.assertRaises(ValueError):
            self.score([(0, 0.5, 0)])

    def test_committed_artifact(self):
        result = summarize()
        saved = json.loads((ROOT / 'output/crnn_model_results_5/verified_summary.json').read_text())
        self.assertEqual(result, saved)
        self.assertEqual(result['confusion_matrix_actual_by_predicted'], [[1816, 64], [24, 21]])

    def test_missing_keras_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/test_model_load.py'), str(Path(directory) / 'absent.keras')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('none is bundled', result.stdout)

    def test_release_excludes_research_data_and_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = build_release(Path(directory) / 'release.zip')
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                self.assertIn('app.py', names)
                self.assertIn('output/crnn_model_results_5/verified_summary.json', names)
                self.assertFalse(any(name.startswith(('data/', '.git/')) for name in names))
                self.assertFalse(any(name.endswith(('.keras', '.pt', '.wav', '.tar.gz')) for name in names))
                self.assertNotIn('Corona-Hack-Respiratory-Sound-Metadata.csv', names)
            with self.assertRaises(FileExistsError):
                build_release(path)


if __name__ == '__main__':
    unittest.main()
