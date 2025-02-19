# -*- coding: utf-8 -*-
import json
import logging
import mteb
from mteb import MTEB

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    tasks = mteb.get_tasks(tasks=["ArguAna"], languages=["eng"])

    evaluation = MTEB(tasks=tasks)
    model_kwargs = {}
    with open("../test_config/mteb/task_to_instructions.json", "r") as f:
        task_to_instructions = json.load(f)
    model_kwargs["task_to_instructions"] = task_to_instructions

    model = mteb.get_model("xxxxx", **model_kwargs) # Same name as defined in llm2vec_models.py

    evaluation.run(model, output_folder="xxxxx",eval_splits=["test"])





