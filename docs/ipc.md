# Python/Go IPC

- Python functions can write messages to the websocket server for distribution

## Address
```
/tmp/mynah.sock
```

## Format
- The first `36` characters of the message should be the user uuid
- The remaining data is transmitted directly to the websocket and distributed to the client
