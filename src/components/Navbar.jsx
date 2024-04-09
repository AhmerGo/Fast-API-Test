import React from "react";
import { Link } from "react-router-dom";

function Navbar() {
  return (
    <nav className="navbar bg-purple p-4 shadow-md">
      <div className="container mx-auto flex justify-center">
        <Link
          to="/"
          className="font-semibold text-white hover:text-gold mx-4 lg:mx-10 transition duration-300 transform hover:-translate-y-1"
        >
          Home
        </Link>
        <Link
          to="/about"
          className="font-semibold text-white hover:text-yellow-500 mx-4 lg:mx-10 transition duration-300 transform hover:-translate-y-1"
        >
          About
        </Link>
        {/* Add other navigation links as needed */}
      </div>
    </nav>
  );
}

export default Navbar;
