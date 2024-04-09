import React from "react";
import logo from "../assets/gpt.svg"; // Path to the logo image
import logoTwo from "../assets/logoo.svg";

function Header() {
  return (
    <header
      className="header bg-white text-purple-600 flex items-center" // Flex applied here
      style={{ height: "100px" }}
    >
      <div className="container mx-auto flex items-center justify-evenly">
        <img
          src={logoTwo}
          alt="Hardin-Simmons Logo"
          className="h-full ml-12" // Adjusted for vertical centering
          style={{ maxWidth: "200px", marginLeft: "65px" }} // Adjust logo width to fit in the header
        />
        <img
          src={logo}
          alt="GPT Logo"
          className="h-full ml-12" // Adjusted for vertical centering
          style={{ maxWidth: "200px", marginLeft: "65px" }} // Adjust logo width to fit in the header
        />
      </div>
    </header>
  );
}

export default Header;
