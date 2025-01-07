# Witcher_Data_Scrapper.py
import yt_dlp
import os
from concurrent.futures import ThreadPoolExecutor
import time
import random
from collections import defaultdict
import json

class Witcher3Scraper:
    def __init__(self, output_dir="witcher3_dataset", max_size_gb=70):
        self.output_dir = output_dir
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.current_size = 0
        self.download_history = defaultdict(list)
        self.search_queries = {
            'boss_fights': [
                'Witcher 3 Griffin boss fight',
                'Witcher 3 Imlerith boss fight',
                'Witcher 3 Eredin boss fight',
                'Witcher 3 Caranthir boss fight',
                'Witcher 3 Toad Prince boss fight'
            ],
            'main_quests': [
                'Witcher 3 Bloody Baron quest',
                'Witcher 3 Ladies of the Wood quest',
                'Witcher 3 Battle of Kaer Morhen',
                'Witcher 3 Final Mission'
            ],
            'gameplay': [
                'Witcher 3 combat gameplay',
                'Witcher 3 signs gameplay',
                'Witcher 3 alchemy gameplay'
            ]
        }
        
        self.ydl_opts = {
            'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/mp4',
            'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'ignoreerrors': True
        }
        
        os.makedirs(output_dir, exist_ok=True)

    def _get_random_delay(self):
        """Generate random delay between requests to avoid rate limiting"""
        return random.uniform(1, 3)

    def _get_video_info(self, url):
        """Extract video information using yt-dlp"""
        with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
            try:
                return ydl.extract_info(url, download=False)
            except Exception as e:
                print(f"Error extracting info for {url}: {str(e)}")
                return None

    def _download_video(self, video_info, category):
        """Download a single video if it meets size constraints"""
        if not video_info:
            return False

        estimated_size = video_info.get('filesize_approx', 0)
        if self.current_size + estimated_size > self.max_size_bytes:
            return False

        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                ydl.download([video_info['webpage_url']])
                
            actual_size = os.path.getsize(f"{self.output_dir}/{video_info['title']}.mp4")
            self.current_size += actual_size
            self.download_history[category].append({
                'title': video_info['title'],
                'url': video_info['webpage_url'],
                'size': actual_size
            })
            return True
            
        except Exception as e:
            print(f"Error downloading {video_info['webpage_url']}: {str(e)}")
            return False

    def _search_and_download(self, query, category):
        """Search for videos and download them"""
        search_opts = self.ydl_opts.copy()
        search_opts['extract_flat'] = True
        search_url = f"ytsearch50:{query}"

        with yt_dlp.YoutubeDL(search_opts) as ydl:
            try:
                results = ydl.extract_info(search_url, download=False)
                if not results:
                    return

                for entry in results['entries']:
                    if not entry:
                        continue
                        
                    time.sleep(self._get_random_delay())
                    video_info = self._get_video_info(entry['url'])
                    
                    if self._download_video(video_info, category):
                        print(f"Downloaded: {video_info['title']}")
                    
                    if self.current_size >= self.max_size_bytes:
                        return

            except Exception as e:
                print(f"Error in search and download for {query}: {str(e)}")

    def scrape(self):
        """Main scraping function"""
        with ThreadPoolExecutor(max_workers=3) as executor:
            for category, queries in self.search_queries.items():
                for query in queries:
                    if self.current_size >= self.max_size_bytes:
                        break
                    executor.submit(self._search_and_download, query, category)
                    time.sleep(self._get_random_delay())

        # Save download history
        with open(f"{self.output_dir}/download_history.json", 'w') as f:
            json.dump(dict(self.download_history), f, indent=4)

        print(f"\nScraping completed!")
        print(f"Total size: {self.current_size / (1024*1024*1024):.2f} GB")
        for category, videos in self.download_history.items():
            print(f"\n{category}: {len(videos)} videos")

if __name__ == "__main__":
    scraper = Witcher3Scraper(max_size_gb=70)
    scraper.scrape()