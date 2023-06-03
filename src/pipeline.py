from src.constants import *
from src.models.train_model import cnn, train_model
from src.mlflow.mlflow_experiment import initiate_mlflow_experiment
from src.features.build_features import build_features
from src.models.evaluate_model import evaluate_model

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

from src.models.train_model import cnn
from src.visualization.visualize import plot_history

# Setup OpenTelemetry
# provider = TracerProvider()
# processor = BatchSpanProcessor(ConsoleSpanExporter())
# provider.add_span_processor(processor)
# trace.set_tracer_provider(provider)
# tracer = trace.get_tracer(__name__)

if __name__ == '__main__':
    # with tracer.start_as_current_span("main"):
        initiate_mlflow_experiment(EXPERIMENT)
        training_dataset, validation_dataset, target_classes = build_features()
        model = cnn(IMG_SIZE, len(target_classes))
        trained_model, history = train_model(model, training_dataset, validation_dataset,  EPOCHS)
        evaluate_model(trained_model, validation_dataset)
        plot_history(history)