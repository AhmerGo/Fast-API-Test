import React from "react";
import { useState } from "react";
import Logo from "../assets/logoo.svg";
export default function Form() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  function handleForm(event) {
    event.preventDefault();
    if (username !== "admin") {
      alert("Username not recognized");
    }
    if (password !== "password") {
      alert("Password not recognized");
    }
  }

  const getInputClassName = (value) => {
    // If input value is empty, apply placeholder-centered style
    if (value === "") {
      return "w-full my-3 border border-black rounded-md placeholder-center text-center";
    }
    // If input value is not empty, apply text-left style
    return "w-full my-3 border border-black rounded-md text-left";
  };

  return (
    <div className="flex flex-wrap justify-center my-14">
      <form onSubmit={handleForm}>
        <img src={Logo} alt="HSU Logo" />
        <input
          type="text"
          name="username"
          id="username"
          placeholder="Username"
          className={getInputClassName(username)}
          onChange={(e) => setUsername(e.target.value)}
        />
        <input
          type="password"
          name="password"
          id="password"
          placeholder="Password"
          onChange={(e) => setPassword(e.target.value)}
          className={getInputClassName(password)}
        />
        <br />
        <section className="flex flex-wrap justify-evenly">
          <button type="submit">Submit</button>
          <a href="forgotPassword.html">Forgot Password?</a>
        </section>
      </form>
    </div>
  );
}
