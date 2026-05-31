import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config_utils import get_path, get_setting, load_project_config


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train transfer learning image classifier.")
	parser.add_argument("--config", type=Path, default=Path("config/project_config.json"))
	parser.add_argument("--processed-dir", type=Path, default=None)
	parser.add_argument("--models-dir", type=Path, default=None)
	parser.add_argument("--figures-dir", type=Path, default=None)
	parser.add_argument("--img-size", type=int, default=None)
	parser.add_argument("--batch-size", type=int, default=None)
	parser.add_argument("--epochs", type=int, default=None)
	parser.add_argument("--learning-rate", type=float, default=None)
	parser.add_argument("--dropout", type=float, default=None)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument(
		"--use-class-weights",
		action="store_true",
		help="Enable balanced class weighting during training.",
	)
	return parser.parse_args()


def resolve_settings(args: argparse.Namespace) -> argparse.Namespace:
	config = load_project_config(args.config)

	args.processed_dir = args.processed_dir or get_path(config, "processed_dir", "data/processed")
	args.models_dir = args.models_dir or get_path(config, "models_dir", "models")
	args.figures_dir = args.figures_dir or get_path(config, "figures_dir", "Figures")

	args.img_size = args.img_size or int(get_setting(config, "training", "img_size", 224))
	args.batch_size = args.batch_size or int(get_setting(config, "training", "batch_size", 16))
	args.epochs = args.epochs or int(get_setting(config, "training", "epochs", 25))
	args.learning_rate = args.learning_rate or float(
		get_setting(config, "training", "learning_rate", 1e-4)
	)
	args.dropout = args.dropout or float(get_setting(config, "training", "dropout", 0.3))
	args.seed = args.seed or int(get_setting(config, "training", "seed", 42))

	use_weights_from_config = bool(get_setting(config, "training", "use_class_weights", False))
	args.use_class_weights = args.use_class_weights or use_weights_from_config
	return args


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)


def build_model(img_size: int, num_classes: int, dropout: float) -> tf.keras.Model:
	base_model = MobileNetV2(
		include_top=False,
		weights="imagenet",
		input_shape=(img_size, img_size, 3),
	)
	base_model.trainable = False

	model = models.Sequential(
		[
			layers.Input(shape=(img_size, img_size, 3)),
			base_model,
			layers.GlobalAveragePooling2D(),
			layers.Dropout(dropout),
			layers.Dense(num_classes, activation="softmax"),
		]
	)
	return model


def compute_weights_from_generator(train_gen) -> dict:
	classes = np.unique(train_gen.classes)
	weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_gen.classes)
	return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def plot_training_curves(history: tf.keras.callbacks.History, out_path: Path) -> None:
	out_path.parent.mkdir(parents=True, exist_ok=True)
	plt.figure(figsize=(10, 4))

	plt.subplot(1, 2, 1)
	plt.plot(history.history.get("accuracy", []), label="train_acc")
	plt.plot(history.history.get("val_accuracy", []), label="val_acc")
	plt.title("Accuracy")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.legend()

	plt.subplot(1, 2, 2)
	plt.plot(history.history.get("loss", []), label="train_loss")
	plt.plot(history.history.get("val_loss", []), label="val_loss")
	plt.title("Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.legend()

	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	plt.close()


def main() -> None:
	args = resolve_settings(parse_args())
	set_seed(args.seed)

	train_dir = args.processed_dir / "train"
	val_dir = args.processed_dir / "val"

	if not train_dir.exists() or not val_dir.exists():
		raise FileNotFoundError(
			"Processed train/val directories not found. "
			"Run scripts/data_preprocessing.py first."
		)

	args.models_dir.mkdir(parents=True, exist_ok=True)
	args.figures_dir.mkdir(parents=True, exist_ok=True)

	train_datagen = ImageDataGenerator(
		preprocessing_function=preprocess_input,
		rotation_range=20,
		width_shift_range=0.1,
		height_shift_range=0.1,
		horizontal_flip=True,
	)
	val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

	train_gen = train_datagen.flow_from_directory(
		str(train_dir),
		target_size=(args.img_size, args.img_size),
		batch_size=args.batch_size,
		class_mode="categorical",
		shuffle=True,
		seed=args.seed,
	)
	val_gen = val_datagen.flow_from_directory(
		str(val_dir),
		target_size=(args.img_size, args.img_size),
		batch_size=args.batch_size,
		class_mode="categorical",
		shuffle=False,
	)

	model = build_model(args.img_size, train_gen.num_classes, args.dropout)
	model.compile(
		optimizer=optimizers.Adam(learning_rate=args.learning_rate),
		loss="categorical_crossentropy",
		metrics=["accuracy"],
	)

	checkpoint_path = args.models_dir / "best_model.keras"
	callback_list = [
		callbacks.ModelCheckpoint(
			filepath=str(checkpoint_path),
			monitor="val_accuracy",
			mode="max",
			save_best_only=True,
			verbose=1,
		),
		callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
		callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7),
	]

	class_weights = compute_weights_from_generator(train_gen) if args.use_class_weights else None

	history = model.fit(
		train_gen,
		validation_data=val_gen,
		epochs=args.epochs,
		callbacks=callback_list,
		class_weight=class_weights,
		verbose=1,
	)

	final_model_path = args.models_dir / "trained_model.h5"
	model.save(final_model_path)

	history_path = args.models_dir / "training_history.json"
	with history_path.open("w", encoding="utf-8") as f:
		json.dump(history.history, f, indent=2)

	class_indices_path = args.models_dir / "class_indices.json"
	with class_indices_path.open("w", encoding="utf-8") as f:
		json.dump(train_gen.class_indices, f, indent=2)

	plot_training_curves(history, args.figures_dir / "training_curves.png")

	print(f"Saved model: {final_model_path}")
	print(f"Saved best model checkpoint: {checkpoint_path}")
	print(f"Saved history: {history_path}")
	print(f"Saved class indices: {class_indices_path}")


if __name__ == "__main__":
	main()
