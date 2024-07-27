import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = np.random.normal(size=100)
data = pd.DataFrame(data, columns=['Dist_norm'])
st.write(data)
fig, ax = plt.subplots()
ax.hist(data.Dist_norm)
st.pyplot(fig)