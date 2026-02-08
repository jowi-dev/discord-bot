{ pkgs ? import <nixpkgs> {} }:

pkgs.rustPlatform.buildRustPackage rec {
  pname = "discord-bot";
  version = "0.1.0";

  src = ./.;

  cargoLock = {
    lockFile = ./Cargo.lock;
  };

  nativeBuildInputs = with pkgs; [ pkg-config ];
  buildInputs = with pkgs; [ openssl ];

  meta = with pkgs.lib; {
    description = "Simple Discord bot in Rust";
    homepage = "https://github.com/jowi-dev/discord-bot";
    license = licenses.mit;
  };
}
