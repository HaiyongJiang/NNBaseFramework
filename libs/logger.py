import os
import logging
import logging.config
import yaml


def set_logger(out_dir, cfpath="configs/log.yaml"):
    assert(os.path.exists(cfpath))
    with open(cfpath, "rt") as f:
        config = yaml.safe_load(f.read())
    config["handlers"]["file"]["filename"] = os.path.join(
        out_dir,
        config["handlers"]["file"]["filename"],
    )
    print(config)
    print("logging to %s" % (config["handlers"]["file"]["filename"]))
    logging.config.dictConfig(config)
    global g_logging
    g_logging = logging.getLogger()


g_logging = None
def get_logger():
    global g_logging
    if g_logging is None:
        raise Exception("Error: logger is not initilized")
    return g_logging

