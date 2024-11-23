package dev.jlynx.magical_drones_webapp.exception;

public class UsernameExistsException extends RuntimeException {

    public UsernameExistsException() {
    }

    public UsernameExistsException(String message) {
        super(message);
    }
}
