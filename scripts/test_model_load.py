#!/usr/bin/env python3
"""Check a trusted local Keras checkpoint; this does not validate preprocessing or accuracy."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path, help='Explicit path to a trusted .keras checkpoint')
    parser.add_argument('--features', type=int, default=49, help='Expected feature count (recorded v2 run: 49)')
    args = parser.parse_args()
    if args.model.suffix != '.keras' or not args.model.is_file():
        print('A local .keras checkpoint is required; none is bundled. PyTorch files are incompatible.')
        return 2
    if args.features < 1:
        parser.error('--features must be positive')
    try:
        import numpy as np
        import tensorflow as tf
        model = tf.keras.models.load_model(args.model, compile=False, safe_mode=True)
        if tuple(model.input_shape) != (None, args.features, 1):
            raise ValueError('Checkpoint input shape does not match the declared feature count')
        output = np.asarray(model(np.zeros((1, args.features, 1), dtype=np.float32), training=False))
        if output.shape != (1, 1) or not np.isfinite(output).all() or not ((0 <= output) & (output <= 1)).all():
            raise ValueError('Expected one finite sigmoid output per row')
    except Exception as exc:
        print(f'Checkpoint check failed ({type(exc).__name__}). Verify dependencies, architecture, and feature count.')
        return 3
    print('Keras load and shape check passed. Feature order, scaling, calibration, and accuracy remain unverified.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
