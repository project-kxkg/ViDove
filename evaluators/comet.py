from comet import download_model, load_from_checkpoint
from .abs_evaluator import AbsApiEvaluator
from logging import Logger

class CometEvaluator(AbsApiEvaluator):
    def __init__(self, logger:Logger, model_name:str = "Unbabel/wmt22-comet-da", batch_size:int=8, gpus:int=0, accelerator:str = "auto") -> None:
        super().__init__()
        self.model_name = model_name
        self.logger = logger
        self.batch_size = batch_size
        self.accelerator = accelerator
        self.gpus = gpus
        self.is_loaded = False
        self.model = None

    def load_model(self) -> None:
        model = download_model(self.model_name)
        model = load_from_checkpoint(model)
        self.model = model
        self.is_loaded = True

    def evaluate(self, input:dict) -> str:

        if not self.is_loaded:
            self.load_model()

        model_output = self.model.predict(input, batch_size=self.batch_size, gpus=self.gpus, accelerator=self.accelerator)
        return model_output