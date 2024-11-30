package dev.jlynx.magical_drones_webapp.auth;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.security.core.GrantedAuthority;

/**
 * A custom implementation of the {@link GrantedAuthority} interface.
 */
@Getter @Setter
@NoArgsConstructor
@Entity
@Table(
        name = "authority",
        uniqueConstraints = { @UniqueConstraint(columnNames = "string_value", name = "string_value_unique") }
)
public class Authority implements GrantedAuthority {

    @Id
    @GeneratedValue(strategy = GenerationType.SEQUENCE, generator = "authorityId")
    @SequenceGenerator(name = "authorityId", sequenceName = "authority_id_seq", allocationSize = 1)
    private Long id;

    @Column(name = "string_value", nullable = false)
    private String stringValue;

    public Authority(Role role) {
        this.stringValue = role.getValue();
    }

    @Override
    public String getAuthority() {
        return stringValue;
    }
}
