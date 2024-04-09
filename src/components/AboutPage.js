import React from "react";

function AboutPage() {
  return (
    <div className="min-h-screen bg-hsu bg-cover bg-center flex items-center justify-center p-4">
      <div className="max-w-4xl w-full bg-white/90 backdrop-blur-lg rounded-2xl shadow-xl p-8 m-4 text-gray-800 transition duration-500 hover:scale-105">
        <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-center text-purple mb-6 animate-pulse">
          About HossBot
        </h1>
        <p className="text-md md:text-lg lg:text-xl font-medium text-center p-4 rounded-lg shadow-md transition duration-500 hover:scale-105 hover:bg-purple hover:text-gold">
          HossBot is an AI-powered chatbot designed specifically for Hardin
          Simmons University. Utilizing the cutting-edge GPT4All language model,
          it provides intelligent and tailored conversations to assist students,
          faculty, and staff with various tasks and inquiries.
        </p>
        <div className="flex flex-wrap justify-center gap-4 mt-8">
          <div className="w-full md:w-1/2 lg:w-1/3 p-4 bg-gold rounded-lg shadow-lg transition duration-500 hover:scale-110">
            <h2 className="text-2xl font-semibold mb-4 text-purple transition duration-500 hover:text-white">
              Innovative AI
            </h2>
            <p className="transition duration-500 hover:text-gray-800">
              Built on the GPT4All model, HossBot understands and responds with
              high accuracy and contextual awareness, making campus life and
              learning more accessible and engaging.
            </p>
          </div>
          <div className="w-full md:w-1/2 lg:w-1/3 p-4 bg-gold rounded-lg shadow-lg transition duration-500 hover:scale-110">
            <h2 className="text-2xl font-semibold mb-4 text-purple transition duration-500 hover:text-white">
              Campus Assistant
            </h2>
            <p className="transition duration-500 hover:text-gray-800">
              HossBot assists with academic information, campus services, event
              details, and more, making it an indispensable tool for the Hardin
              Simmons University community.
            </p>
          </div>
        </div>
        <div className="mt-10 text-center">
          <a
            href="/"
            className="inline-block bg-purple text-white font-bold py-3 px-6 rounded-full shadow-lg transition duration-500 hover:bg-indigo-700 hover:scale-105"
          >
            Try HossBot Now
          </a>
        </div>
      </div>
    </div>
  );
}

export default AboutPage;
