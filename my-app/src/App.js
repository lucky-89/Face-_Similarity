import React, { useState, useEffect } from "react";
import * as faceapi from "face-api.js";

function App() {
  const [aadharImage, setAadharImage] = useState(null);
  const [userImage, setUserImage] = useState(null);
  const [similarity, setSimilarity] = useState(null);

  useEffect(() => {
    async function loadModels() {
      await faceapi.nets.ssdMobilenetv1.loadFromUri("/models"); // Face detection model
      await faceapi.nets.faceLandmark68Net.loadFromUri("/models"); // Facial landmarks
      await faceapi.nets.faceRecognitionNet.loadFromUri("/models"); // Face embeddings
    }
    loadModels();
  }, []);

  const handleImageUpload = (event, setImage) => {
    const file = event.target.files[0];
    if (file) {
      const imageUrl = URL.createObjectURL(file);
      setImage(imageUrl);
    }
  };

  const getFaceEmbedding = async (imageSrc) => {
    const img = await faceapi.fetchImage(imageSrc);
    const detection = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
    return detection ? detection.descriptor : null;
  };

  const cosineSimilarity = (vecA, vecB) => {
    const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
    return dotProduct / (Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0)) * Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0)));
  };

  const compareFaces = async () => {
    if (!aadharImage || !userImage) {
      alert("Please upload both images.");
      return;
    }

    const aadharFeatures = await getFaceEmbedding(aadharImage);
    const userFeatures = await getFaceEmbedding(userImage);

    if (aadharFeatures && userFeatures) {
      const sim = cosineSimilarity(aadharFeatures, userFeatures);
      setSimilarity(sim.toFixed(4));
    } else {
      alert("No face detected in one or both images.");
    }
  };

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h1>Aadhaar Face Verification</h1>

      <div>
        <input type="file" accept="image/*" onChange={(e) => handleImageUpload(e, setAadharImage)} />
        {aadharImage && <img src={aadharImage} alt="Aadhaar" width="200" style={{ margin: "10px" }} />}
      </div>

      <div>
        <input type="file" accept="image/*" onChange={(e) => handleImageUpload(e, setUserImage)} />
        {userImage && <img src={userImage} alt="User" width="200" style={{ margin: "10px" }} />}
      </div>

      <button onClick={compareFaces} style={{ marginTop: "20px", padding: "10px 20px", fontSize: "16px" }}>
        Compare Faces
      </button>

      {similarity !== null && (
        <h2 style={{ marginTop: "20px" }}>Similarity Score: {similarity}</h2>
      )}
    </div>
  );
}

export default App;
