from comet import download_model, load_from_checkpoint
from .abs_evaluator import AbsApiEvaluator
from logging import Logger

class CometEvaluator(AbsApiEvaluator):
    def __init__(self, model_name:str, logger:Logger) -> None:
        super().__init__()
        self.model_name = model_name
        self.logger = logger
        self.is_loaded = False
        self.model = None

    def load_model(self) -> None:
        model = download_model(self.model_name)
        model = load_from_checkpoint(model)
        self.model = model
        self.is_loaded = True

    def evaluate(self, input:str) -> str:

        if not self.is_loaded:
            self.load_model()

        model_output = self.model.predict(input, batch_size=8, gpus=1)
        return model_output