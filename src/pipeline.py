from src.constants import *
from src.models.train_model import cnn, train_model
from src.mlflow_setup import setup_mlflow
from src.features.build_features import build_features

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

from src.models.predict_model import evaluate_model
from src.models.train_model import cnn
from src.visualization.visualize import plot_history

# Setup OpenTelemetry
provider = TracerProvider()
processor = BatchSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

if __name__ == '__main__':
    with tracer.start_as_current_span("main"):
        # Setup MLflow
        setup_mlflow(EXPERIMENT_NAME)


        # Preprocess the data
        training_dataset, validation_dataset, target_classes = build_features()

        # Load the model
        model = cnn(IMG_SIZE, len(target_classes))

        # Train the model
        trained_model, history = train_model(model, training_dataset, validation_dataset,  EPOCHS)

        # Evaluate the model
        evaluate_model(model, validation_dataset)

        # Visualize the results
        plot_history(history)