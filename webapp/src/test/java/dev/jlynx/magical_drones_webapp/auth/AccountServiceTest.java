package dev.jlynx.magical_drones_webapp.auth;

import dev.jlynx.magical_drones_webapp.dto.AccountRegistration;
import dev.jlynx.magical_drones_webapp.exception.UsernameExistsException;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.mockito.ArgumentCaptor;
import org.mockito.Captor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.security.crypto.password.PasswordEncoder;

import java.util.stream.Stream;

import static org.mockito.ArgumentMatchers.*;
import static org.mockito.BDDMockito.given;
import static org.mockito.BDDMockito.then;
import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;

@Tag("unit")
@ExtendWith(MockitoExtension.class)
class AccountServiceTest {

    @Mock
    private PasswordEncoder passwordEncoderMock;

    @Mock
    private AccountRepository accountRepositoryMock;

    @Captor
    ArgumentCaptor<Account> accountCaptor;

    @InjectMocks
    private AccountService underTest;

    @Test
    void registerAccount_ShouldThrow_WhenUsernameAlreadyExists() {
        // given
        String username = "someuser";
        AccountRegistration registration = new AccountRegistration(username, "password");
        given(accountRepositoryMock.existsByUsername(anyString())).willReturn(true);

        // when
        Exception thrown = null;
        try {
            underTest.registerAccount(registration);
        } catch (Exception ex) {
            thrown = ex;
        }

        // then
        then(accountRepositoryMock).should().existsByUsername(username);
        then(accountRepositoryMock).should(never()).save(any());
        assertThat(thrown)
                .isNotNull()
                .isInstanceOf(UsernameExistsException.class);
    }

    @ParameterizedTest
    @MethodSource("validRegistrationProvider")
    void registerAccount_ShouldSave_WhenDataValid(AccountRegistration registration) {
        // given
        String encodedPwd = "encoded";
        Account expectedAcc = new Account(registration.username(), encodedPwd);
        expectedAcc.setId(34L);
        given(passwordEncoderMock.encode(anyString())).willReturn(encodedPwd);
        given(accountRepositoryMock.save(any())).willReturn(expectedAcc);
        given(accountRepositoryMock.existsByUsername(anyString())).willReturn(false);

        // when
        Account returned = underTest.registerAccount(registration);

        // then
        then(passwordEncoderMock).should(times(1)).encode(registration.password());
        then(accountRepositoryMock).should(times(1)).save(accountCaptor.capture());
        assertThat(returned)
                .hasFieldOrPropertyWithValue("username", registration.username())
                .hasFieldOrPropertyWithValue("password", encodedPwd)
                .extracting(Account::getId).isNotNull();
        assertThat(accountCaptor.getValue())
                .hasFieldOrPropertyWithValue("username", registration.username())
                .hasFieldOrPropertyWithValue("password", encodedPwd);
    }

    private static Stream<AccountRegistration> validRegistrationProvider() {
        return Stream.of(
                new AccountRegistration("ivy", "P@ssw0rd"),
                new AccountRegistration("john_doe5", "abcDEF123$%^")
        );
    }

    @Test
    void deleteAccount_ShouldDelete_WhenAccountDoesNotExist() {
        // given
        long id = 17L;
        given(accountRepositoryMock.existsById(anyLong())).willReturn(false);

        // when
        boolean returned = underTest.deleteAccount(id);

        // then
        assertThat(returned).isFalse();
    }

    @Test
    void deleteAccount_ShouldDelete_WhenAccountExists() {
        // given
        long id = 17L;
        given(accountRepositoryMock.existsById(anyLong())).willReturn(true);

        // when
        boolean returned = underTest.deleteAccount(id);

        // then
        then(accountRepositoryMock).should(times(1)).deleteById(id);
        assertThat(returned).isTrue();
    }

}