
"""
    Author: julij
    Date: 12/09/2020
    Description:
"""


import inspect
import textwrap
from collections import OrderedDict
import streamlit as st
from streamlit.logger import get_logger
from clo_clustering import run_clo_clustering
from da_price_analytics import run_price_analytics

LOGGER = get_logger(__name__)


def intro():
    st.markdown(
        """
        ### Welcome to Machine Learning and Data Science Page

        **:point_left: Select a model from dropdown on the left** and start exploring...
    """
    )


MODELS = OrderedDict(
    [
        ("", (intro, None)),
        ("Digital Assets Price Analytics", (run_price_analytics, None)),
        ("European CLOs Clustering Model", (run_clo_clustering, None)),
    ]
)


def run():
    model_name = st.sidebar.selectbox("Choose a model", list(MODELS.keys()), 0)
    model = MODELS[model_name][0]
    show_code = False
    if model_name == "—":
        st.write("# Welcome to MLDS page!")
    else:
        # show_code = st.sidebar.checkbox("Show code", True)
        st.markdown("# %s" % model_name)
        description = MODELS[model_name][1]
        if description:
            st.write(description)
        # Clear everything from the intro page.
        # We only have 4 elements in the page so this is intentional overkill.
        for i in range(10):
            st.empty()

    model()

    if show_code:
        st.markdown("## Code")
        sourcelines, _ = inspect.getsourcelines(model)
        st.code(textwrap.dedent("".join(sourcelines[1:])))


if __name__ == "__main__":
    run()



