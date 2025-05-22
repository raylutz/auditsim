import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

freq = st.slider("Frequency", 1, 10, 5)
x = np.linspace(0, 10, 500)
y = np.sin(freq * x)

fig, ax = plt.subplots()
ax.plot(x, y)
st.pyplot(fig)
