package dev.jlynx.magical_drones_webapp.auth;

import dev.jlynx.magical_drones_webapp.generation.Generation;
import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

@Setter
@NoArgsConstructor
@Entity
public class Account implements UserDetails {

    @Getter
    @Id
    @GeneratedValue(strategy = GenerationType.SEQUENCE, generator = "accountId")
    @SequenceGenerator(name = "accountId", sequenceName = "account_id_seq", allocationSize = 50)
    private Long id;

    @Column(nullable = false, unique = true)
    private String username;

    @Column(nullable = false)
    private String password;

    private Boolean accountNonExpired;
    private Boolean accountNonLocked;
    private Boolean credentialsNonExpired;
    private Boolean enabled;

    @ManyToMany(fetch = FetchType.EAGER)
    @JoinTable(
            name = "account_authority",
            joinColumns = @JoinColumn(name = "account_id"),
            inverseJoinColumns = @JoinColumn(name = "authority_id"),
            foreignKey = @ForeignKey(name = "fk_account_authority"),
            inverseForeignKey = @ForeignKey(name = "fk_authority_account")
    )
    private List<Authority> authorities;

    /**
     * An id of the file containing this account's profile picture.
     */
    @Getter
    @Column(name = "profile_picture", unique = true)
    private Long profilePicture;

    @OneToMany(mappedBy = "account", cascade = { CascadeType.ALL }, orphanRemoval = true)
    private List<Generation> generations;


    public Account(String username, String password) {
        this.username = username;
        this.password = password;
        profilePicture = null;
        accountNonExpired = true;
        accountNonLocked = true;
        credentialsNonExpired = true;
        enabled = true;
        authorities = List.of(new Authority(Role.USER));
        generations = new ArrayList<>();
    }

    @Override
    public Collection<? extends GrantedAuthority> getAuthorities() {
        return authorities;
    }

    @Override
    public String getUsername() {
        return username;
    }

    @Override
    public String getPassword() {
        return password;
    }

    @Override
    public boolean isAccountNonExpired() {
        return accountNonExpired;
    }

    @Override
    public boolean isAccountNonLocked() {
        return accountNonLocked;
    }

    @Override
    public boolean isCredentialsNonExpired() {
        return credentialsNonExpired;
    }

    @Override
    public boolean isEnabled() {
        return enabled;
    }

    public void addGeneration(Generation generation) {
        generations.add(generation);
        generation.setAccount(this);
    }

    public boolean removeGeneration(Generation generation) {
        if (generations.contains(generation)) {
            generations.remove(generation);
            generation.setAccount(null);
            return true;
        }
        return false;
    }
}
