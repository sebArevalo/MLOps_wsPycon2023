Proyecto: Verificación de Identidad con LFW y FaceNet

Este proyecto implementa un flujo completo de MLOps para la tarea de verificación de identidad facial utilizando el dataset LFW (Labeled Faces in the Wild) en su variante de pares.

El pipeline incluye:

Carga de datos desde scikit-learn (fetch_lfw_pairs) y división en train/val/test.

Preprocesamiento de imágenes y generación de embeddings faciales con FaceNet (InceptionResnetV1).

Modelo de clasificación (pairhead), que combina las representaciones de cada par mediante:

Diferencia absoluta |e1 − e2|

Producto de Hadamard e1 ⊙ e2

Similitud coseno y (1 − coseno)

Entrenamiento y validación del modelo con PyTorch.

Seguimiento de experimentos y gestión de datasets/modelos con Weights & Biases (W&B).

Automatización con GitHub Actions, que ejecuta las etapas de load, preprocess, build, train y eval en cada cambio del repositorio.

Modificaciones respecto al proyecto base

Se realizaron ajustes ligeros al código original:

Se reemplazó el uso de transformaciones de torchvision por un preprocesamiento puro en PyTorch, para trabajar directamente con tensores y evitar incompatibilidades.

Se integró un registro de modelos en src/model/src para permitir la creación flexible de diferentes arquitecturas de clasificación.

Se adaptaron los workflows de GitHub Actions para correr en modo módulo (python -m ...) y asegurar la correcta resolución de imports.

Resultados

Entrenamiento estable y reproducible en CI/CD.

Métricas de validación y pruebas registradas en W&B.

Repositorio estructurado siguiendo buenas prácticas de MLOps.
