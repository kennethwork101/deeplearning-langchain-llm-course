from kwwutils import clock, printit
from src.deeplearning_langchain_llm_course._1_3_lcel_from_templates import (
    main,
)


@clock
def test_func(options, model):
    printit("options", options)
    printit("model", model)
    options["model"] = model
    options["llm_type"] = "llm"
    response1, response2 = main(**options)
    printit("response1", response1)
    printit("response2", response2)
    assert response1
    assert response2