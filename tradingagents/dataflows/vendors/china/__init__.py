"""China A-share data: prices, statements, news, announcements, forums and macro.

Yahoo carries A-share prices but little else: no filing dates on statements, no
exchange announcements, no Chinese news or forum chatter. This vendor reads
them from Chinese sources through AKShare (East Money, CNINFO, CCTV, Cailian
Press) and, for the East Money stock forum, from its public pages.

Every dated path keeps the point-in-time rules the other vendors follow
(``date_window``): statements are served by the date they were announced,
announcements and posts are trimmed to the analysis window, and a feed that
only serves recent items reports a historical window as unavailable rather than
as silence.

The router sends A-share tickers here first (``china_auto_vendor`` in the
config) and falls back to the configured chain when a source fails.
"""
