package dev.jlynx.magical_drones_webapp.storage;

import dev.jlynx.magical_drones_webapp.exception.StorageException;

/**
 * Defines a set of operations for interacting with cloud storage service providers.
 */
public interface StorageService {

    void upload(String bucketName, String key, byte[] payload) throws StorageException;
    byte[] download(String bucketName, String key) throws StorageException;
    void delete(String bucketName, String key) throws StorageException;
}
