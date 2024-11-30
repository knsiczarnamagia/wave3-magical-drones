package dev.jlynx.magical_drones_webapp.storage.s3;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

@Getter @Setter
@Configuration
@ConfigurationProperties(prefix = "aws")
public class AwsProperties {

    private String region;
    private S3Properties s3;

    @Getter @Setter
    public static class S3Properties {
        private String bucket;
        private String bucketTest;
    }
}
