{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "03a4876b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import numpy as np\n",
    "from tensorflow.keras.models import load_model\n",
    "from tensorflow.keras.preprocessing import image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "995d87b1",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.\n"
     ]
    }
   ],
   "source": [
    "model = load_model('fire_detection_model.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "4f9c1863",
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict(image_file):\n",
    "    img = image.load_img(image_file, target_size=(64, 64))\n",
    "    img = image.img_to_array(img)\n",
    "    img = np.expand_dims(img, axis=0)\n",
    "    prediction = model.predict(img)\n",
    "    return prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "c289b452",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-03-30 13:21:23.625 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\LENOVO\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "def main():\n",
    "    st.title('Fire Detection App')\n",
    "\n",
    "    # File uploader\n",
    "    uploaded_file = st.file_uploader(\"Choose an image...\", type=[\"jpg\", \"jpeg\", \"png\"])\n",
    "\n",
    "    if uploaded_file is not None:\n",
    "        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)\n",
    "        st.write(\"\")\n",
    "\n",
    "        # Make prediction\n",
    "        prediction = predict(uploaded_file)\n",
    "        if prediction[0][0] == 1:\n",
    "            result = 'Fire'\n",
    "        elif prediction[0][0] == 0:\n",
    "            result = 'Non-Fire'\n",
    "        else:\n",
    "            result = 'smoke'\n",
    "        \n",
    "        st.write(\"Prediction:\", result)\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9d60ca66",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
