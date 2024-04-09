import React from "react";
import { Routes, Route } from "react-router-dom";
import Header from "./components/Header";
import ChatBot from "./components/ChatBot";
import Navbar from "./components/Navbar";
import AboutPage from "./components/AboutPage";

function App() {
  return (
    <div className="bg-purple-200 min-h-screen flex flex-col">
      <Header />
      <Navbar />
      <Routes>
        <Route
          path="/"
          element={
            <div className="bg-white rounded-lg shadow-lg p-6 m-4">
              <h1 className="text-4xl font-extrabold text-purple mb-6 text-center">
                Welcome to the HossBot Chat Interface
              </h1>
              <p className="text-gray-800 text-xl mb-8 text-center">
                Here you can interact with our AI chatbot.
              </p>
              <ChatBot />
            </div>
          }
        />
        <Route path="/about" element={<AboutPage />} />
      </Routes>
    </div>
  );
}

export default App;
