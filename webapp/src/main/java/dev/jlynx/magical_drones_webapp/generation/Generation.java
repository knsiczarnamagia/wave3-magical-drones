package dev.jlynx.magical_drones_webapp.generation;

import dev.jlynx.magical_drones_webapp.auth.Account;
import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;
import java.util.Objects;

@Getter @Setter
@NoArgsConstructor
@Entity
public class Generation {

    @Id
    @GeneratedValue(strategy = GenerationType.SEQUENCE, generator = "generationId")
    @SequenceGenerator(name = "generationId", sequenceName = "generation_id_seq", allocationSize = 50)
    private Long id;

    /**
     * Datetime in UTC when the generation request was submitted.
     */
    @Column(name = "started_at")
    private LocalDateTime startedAt;

    /**
     * Datetime in UTC when the generation process was finished.
     */
    @Column(name = "completed_at")
    private LocalDateTime completedAt;

    /**
     * An id of the source image file.
     */
    @Column(name = "source_image", unique = true)
    private Long sourceImage;

    /**
     * An id of the target image file.
     */
    @Column(name = "generated_image", unique = true)
    private Long generatedImage;

    @ManyToOne
    @JoinColumn(name = "account_id", foreignKey = @ForeignKey(name = "fk_generation_account"))
    private Account account;

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Generation that = (Generation) o;
        return Objects.equals(id, that.id);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(id);
    }
}
