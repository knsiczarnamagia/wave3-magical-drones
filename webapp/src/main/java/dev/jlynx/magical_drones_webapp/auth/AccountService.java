package dev.jlynx.magical_drones_webapp.auth;

import dev.jlynx.magical_drones_webapp.dto.AccountRegistration;
import dev.jlynx.magical_drones_webapp.exception.UsernameExistsException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

@Service
public class AccountService {

    private static final Logger log = LoggerFactory.getLogger(AccountService.class);

    private final AccountRepository accountRepository;
    private final PasswordEncoder encoder;

    @Autowired
    public AccountService(AccountRepository accountRepository, PasswordEncoder encoder) {
        this.accountRepository = accountRepository;
        this.encoder = encoder;
    }

    /**
     * Registers a new user account if the username does not already exist.
     *
     * @param reg a DTO object containing the data for the new account
     * @return the persisted {@code Account} entity
     * @throws UsernameExistsException if an account with the given username already exists
     */
    public Account registerAccount(AccountRegistration reg) {
        if (accountRepository.existsByUsername(reg.username())) {
            log.debug("Tried to register account with existing username: {}", reg.username());
            throw new UsernameExistsException(String.format("Username '%s' already exists.", reg.username()));
        }
        Account newAccount = new Account(
                reg.username(),
                encoder.encode(reg.password())
        );
        Account saved = accountRepository.save(newAccount);
        log.debug("Registered account with username='{}' and id={}", saved.getUsername(), saved.getId());
        return saved;
    }

    /**
     * Deletes an account by its ID if it exists.
     *
     * @param id the ID of the account to delete
     * @return {@code true} if the account was deleted, {@code false} if no account with the given ID exists
     */
    public boolean deleteAccount(long id) {
        if (!accountRepository.existsById(id)) {
            log.debug("Tried to delete an account with non-existing id={}", id);
            return false;
        }
        accountRepository.deleteById(id);
        log.debug("Deleted account with id={}", id);
        return true;
    }
}
