{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "aa858abc",
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
   "execution_count": 5,
   "id": "56ed6b20",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\LENOVO\\anaconda3\\Lib\\site-packages\\keras\\src\\saving\\saving_lib.py:396: UserWarning: Skipping variable loading for optimizer 'rmsprop', because it has 14 variables whereas the saved optimizer has 26 variables. \n",
      "  trackable.load_own_variables(weights_store.get(inner_path))\n"
     ]
    }
   ],
   "source": [
    "model = load_model('fire_detection_model.keras')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "58646072",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "4dc65b51",
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
   "execution_count": 11,
   "id": "778d6579",
   "metadata": {},
   "outputs": [],
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
   "execution_count": none,
   "id": "7e300b24",
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
