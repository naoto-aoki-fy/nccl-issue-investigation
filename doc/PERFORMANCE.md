# Send/Receive performance

`sendrecv_perf` reports operation time in microseconds and bandwidth in GB/s
(decimal gigabytes). Each rank sends to the next rank and receives from the
previous rank in a ring.

For a message containing `count` elements of `typesize` bytes and measured time
`seconds`, the reported bandwidth is:

```
algbw = count * typesize / seconds / 1e9
busbw = algbw
```

The bandwidth uses the size of the sent message; it does not add sent and received
bytes together. Send/Receive uses a bus bandwidth correction factor of 1.
In-place Send/Receive is not supported, and its data validation is disabled.
