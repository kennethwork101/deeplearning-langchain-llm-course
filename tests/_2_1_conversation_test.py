""" 
"""

from kwwutils import clock, printit
from src.deeplearning_langchain_llm_course._2_conversation_bad import (
    main,
)


@clock
def test_func(options, model):
    printit("options", options)
    printit("model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    response = main(**options)
    printit("response", response)
    assert response["input"] == "What is 1+2 and report the result as an integer?"
    assert "3" in response["response"]