package org.apache.calcite.rel.externalize;

import org.apache.calcite.util.TimestampString;
import org.junit.Ignore;
import org.junit.Test;

import java.util.Random;
import java.util.function.BiConsumer;
import java.util.function.LongSupplier;

import static org.apache.calcite.rel.externalize.MapDRelJson.longToTimestampString;
import static org.apache.calcite.rel.externalize.MapDRelJson.timestampStringToLong;
import static org.junit.Assert.assertEquals;

public class MapDRelJsonTest {
    private static final long maxSeconds = new TimestampString("9999-12-31 23:59:59").getMillisSinceEpoch() / 1000;
    private static final long[] MAX_VALUES = new long[]{ // Maximum value for each precision
            maxSeconds,
            maxSeconds * 1000 + 999,
            maxSeconds * 1000_000 + 999999,
            0x7FFFFFFFFFFFFFFFL,
    };
    private static final int[] PRECISIONS = new int[]{0, 3, 6, 9};

    @Test
    public void testLongToTimestampString() {
        benchOrTestLongToTimestampString(MapDRelJsonTest::testLongToTimestampString);
    }

    @Test
    @Ignore
    public void benchLongToTimestampString() {
        benchOrTestLongToTimestampString(MapDRelJsonTest::benchLongToTimestampString);
    }

    private static void benchOrTestLongToTimestampString(BiConsumer<Long, Integer> func) {
        for (int i = 0; i < PRECISIONS.length; i++) {
            int precision = PRECISIONS[i];
            func.accept(0L, precision);
            func.accept(1L, precision);
            func.accept(MAX_VALUES[i], precision);

            for (int j = 0; j <= precision; j++) {
                func.accept((long) Math.pow(10, j), precision);
            }

            long max = MAX_VALUES[i];
            Random rnd = new Random();
            for (int j = 0; j < 10; j++) {
                func.accept(rnd.nextLong() & max, precision);
            }
        }
    }

    private static void testLongToTimestampString(long value, int precision) {
        TimestampString ts;
        switch (precision) {
            case 0:
                ts = TimestampString.fromMillisSinceEpoch(value * 1000);
                break;
            case 3:
                ts = TimestampString.fromMillisSinceEpoch(value);
                break;
            case 6:
                long seconds = value / 1000_000;
                int nanos = (int) (value - (seconds * 1000_000)) * 1000;
                ts = TimestampString.fromMillisSinceEpoch(seconds * 1000).withNanos(nanos);
                break;
            case 9:
                seconds = value / 1000_000_000;
                nanos = (int) (value - (seconds * 1000_000_000));
                ts = TimestampString.fromMillisSinceEpoch(seconds * 1000).withNanos(nanos);
                break;
            default:
                throw new IllegalArgumentException();
        }

        TimestampString ltts = longToTimestampString(value, precision);
        assertEquals(ts.toString(), ltts.toString());
        assertEquals(value, timestampStringToLong(ltts, precision));
    }

    private static void benchLongToTimestampString(long value, int precision) {
        System.out.print("Value=" + value + ", precision=" + precision);

        int nRepeat = 100000;
        int nWarmUp = 0;
        LongSupplier ltts = () -> timestampStringToLong(longToTimestampString(value, precision), precision);
        LongSupplier ts;
        switch (precision) {
            case 0:
                ts = () -> TimestampString.fromMillisSinceEpoch(value * 1000).getMillisSinceEpoch() / 1000;
                break;
            case 3:
                ts = () -> TimestampString.fromMillisSinceEpoch(value).getMillisSinceEpoch();
                break;
            case 6:
                ts = () -> {
                    long seconds = value / 1000_000;
                    int nanos = (int) (value - (seconds * 1000_000)) * 1000;
                    long v = TimestampString.fromMillisSinceEpoch(seconds * 1000).withNanos(nanos)
                            .getMillisSinceEpoch() * 1000;
                    assertEquals(value / 1000 * 1000, v);
                    return value;
                };
                break;
            case 9:
                ts = () -> {
                    long seconds = value / 1000_000_000;
                    int nanos = (int) (value - (seconds * 1000_000_000));
                    long v = TimestampString.fromMillisSinceEpoch(seconds * 1000).withNanos(nanos)
                            .getMillisSinceEpoch() * 1000_000;
                    assertEquals(value / 1000_000 * 1000_000, v);
                    return value;
                };
                break;
            default:
                throw new IllegalArgumentException();
        }

        for (int i = 0; i < nWarmUp; i++) {
            ts.getAsLong();
            ltts.getAsLong();
        }

        long tsTime = System.currentTimeMillis();
        for (int i = 0; i < nRepeat; i++) {
            long v = ts.getAsLong();
            assertEquals(value, v);
        }
        tsTime = System.currentTimeMillis() - tsTime;

        long lttsTime = System.currentTimeMillis();
        for (int i = 0; i < nRepeat; i++) {
            long v = ltts.getAsLong();
            assertEquals(value, v);
        }
        lttsTime = System.currentTimeMillis() - lttsTime;

        System.out.println(", tsTime=" + tsTime + ", lttsTime=" + lttsTime);
    }
}
