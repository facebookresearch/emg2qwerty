import os
from dotenv import load_dotenv


try:
    get_ipython().run_line_magic("load_ext", "autoreload")  # noqa
    get_ipython().run_line_magic("autoreload", "2")  # noqa

    get_ipython().run_line_magic("config", 'InlineBackend.figure_format = "jpeg"')  # noqa
    get_ipython().run_line_magic("matplotlib", "inline")  # noqa

    import matplotlib

    matplotlib.pyplot.style.use("default")
    matplotlib.rcParams["figure.dpi"] = 72
except NameError as ex:
    print("Could not load magic extensions:", ex)


if os.getcwd().endswith("research"):
    os.chdir("..")
    print("Working on", os.getcwd())


print("Loaded .env file:", load_dotenv(".env"))