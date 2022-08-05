from impl.services.modules.core.resources import *

def make_model_from_string(model_type : str, target_classes : int) -> nn.Module:
    MODEL_LIST = ["resnet50"]#, "resnet18", "densenet202"]
    MODEL_MAP = {
        "resnet18" : "AutoResnet18",
        "resnet50" : "AutoResnet",
        "densenet202" : "AutoDensenet202"
    }

    if model_type not in MODEL_LIST:
        raise ReiformTrainingException("Model not available")

    model_classname : str = MODEL_MAP[model_type]

    model = eval("{}({})".format(model_classname, target_classes))
    return model

def load_reiform_model(model_path : str, model_type : str, target_classes : int) -> nn.Module:
    
    model : nn.Module = make_model_from_string(model_type, target_classes)

    # Load the state dict
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model