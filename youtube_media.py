"""YouTube metadata lookup only. No Gemini, downloads, browser cookies or login."""
from dataclasses import dataclass
import re


class MediaLookupError(RuntimeError):
    pass


@dataclass(frozen=True)
class Video:
    video_id: str
    title: str


class QuietLogger:
    # yt-dlp can otherwise write directly to the console from its worker.
    def debug(self, message): pass
    def warning(self, message): pass
    def error(self, message): pass


def find_video(query):
    """Return the first public video search match without extracting media URLs."""
    try:
        from yt_dlp import YoutubeDL
        options = {
            'extract_flat': True, 'skip_download': True, 'quiet': True,
            'no_warnings': True, 'noprogress': True, 'cachedir': False,
            'socket_timeout': 4, 'retries': 0, 'extractor_retries': 0,
            'fragment_retries': 0, 'playlistend': 1, 'lazy_playlist': True,
            'logger': QuietLogger(),
        }
        with YoutubeDL(options) as client:
            results = client.extract_info('ytsearch1:' + query, download=False)
            entry = next(iter((results or {}).get('entries') or []), None)
            video_id = entry.get('id', '') if entry else ''
            if not isinstance(video_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{11}', video_id):
                raise MediaLookupError('No valid video result')
            title = ''.join(c for c in str(entry.get('title') or query) if c.isprintable())[:160]
            return Video(video_id, title)
    except MediaLookupError:
        raise
    except Exception as exc:
        # Caller logs only the exception type, never request bodies or URLs.
        raise MediaLookupError('YouTube search unavailable') from exc
