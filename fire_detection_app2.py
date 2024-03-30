{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "e67b9b98",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "from PIL import Image\n",
    "import numpy as np\n",
    "from tensorflow.keras.models import load_model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "0d155bcb",
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_detection_model():\n",
    "    model = load_model('fire_detection_model.h5')\n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "30df4113",
   "metadata": {},
   "outputs": [],
   "source": [
    "def preprocess_image(image):\n",
    "    # Resize image to target size\n",
    "    image = image.resize((224, 224))\n",
    "    # Convert image to array and normalize pixel values\n",
    "    image = np.asarray(image) / 255.0\n",
    "    # Expand dimensions to match model input shape (batch size = 1)\n",
    "    image = np.expand_dims(image, axis=0)\n",
    "    return image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "84832c7d",
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict_fire_detection(image, model):\n",
    "    prediction = model.predict(image)\n",
    "    return prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "7c4d19b3",
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
    "def main():\n",
    "    st.title('Fire Detection Model')\n",
    "\n",
    "    # Upload image\n",
    "    uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])\n",
    "\n",
    "    # Load model\n",
    "    model = load_detection_model()\n",
    "\n",
    "    # Display image and prediction\n",
    "    if uploaded_image is not None:\n",
    "        # Preprocess and display uploaded image\n",
    "        image = Image.open(uploaded_image)\n",
    "        st.image(image, caption='Uploaded Image', use_column_width=True)\n",
    "\n",
    "        # Preprocess image for prediction\n",
    "        processed_image = preprocess_image(image)\n",
    "\n",
    "        # Make prediction\n",
    "        prediction = predict_fire_detection(processed_image, model)\n",
    "\n",
    "        # Display prediction results\n",
    "        st.subheader('Prediction Results')\n",
    "        st.write(f'Fire Probability: {prediction[0][0]}')\n",
    "        st.write(f'Smoke Probability: {prediction[0][1]}')\n",
    "        st.write(f'Non-fire Probability: {prediction[0][2]}')\n",
    "\n",
    "# Run the Streamlit app\n",
    "if __name__ == '__main__':\n",
    "    main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9fc61c9a",
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
