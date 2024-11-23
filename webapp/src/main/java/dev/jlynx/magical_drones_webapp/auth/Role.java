package dev.jlynx.magical_drones_webapp.auth;

/**
 * Defines valid user roles and their string values.
 * <p>
 * The string value of this enum should be used to define {@code UserAuthority} values.
 * <p>
 * Example:
 * <pre>{@code
 * UserAuthority authority = new UserAuthority(Role.USER);
 * }</pre>
 *
 * @see Authority
 */
public enum Role {

    USER("ROLE_USER"), ADMIN("ROLE_ADMIN");

    private final String value;

    Role(String value) {
        this.value = value;
    }

    public String getValue() {
        return value;
    }
}
