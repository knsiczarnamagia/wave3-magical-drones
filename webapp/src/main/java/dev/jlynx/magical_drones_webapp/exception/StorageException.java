package dev.jlynx.magical_drones_webapp.exception;

import dev.jlynx.magical_drones_webapp.storage.StorageService;

/**
 * A checked when an error has occurred while interacting with a {@link StorageService}.
 */
public class StorageException extends Exception {

    public StorageException() {
    }

    public StorageException(String message) {
        super(message);
    }

    public StorageException(String message, Throwable cause) {
        super(message, cause);
    }
}
