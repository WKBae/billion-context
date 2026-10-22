/** Pseudo-origin of the inline lane (#3278). Never DNS-resolved: any request
 *  reaching this host is served in-process by installInlineTransport before
 *  egress, so it is unreachable from any other process. */
export const INLINE_ORIGIN = "http://billion-context.inline";
