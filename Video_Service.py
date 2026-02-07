"""
Video generation service - integrates NVIDIA Riva TTS and video generation
"""
import asyncio
import httpx
from typing import Dict, Optional, List
from google.cloud import storage, pubsub_v1
import logging
import json
import os
from datetime import datetime

from app.core.config import settings

logger = logging.getLogger(__name__)


class VideoGenerationService:
    """Service for generating motivational videos with AI"""
    
    def __init__(self):
        self.gcs_client = storage.Client()
        self.video_bucket = self.gcs_client.bucket(settings.GCS_VIDEO_BUCKET)
        self.did_api_key = settings.DID_API_KEY
        self.did_url = settings.DID_API_URL
        self.publisher = pubsub_v1.PublisherClient()
    
    async def generate_video(
        self, 
        script: str, 
        clip_id: str,
        voice_name: Optional[str] = None,
        presenter_image: Optional[str] = None
    ) -> Dict:
        """
        Generate complete video from script
        
        Args:
            script: Text script for the video
            clip_id: Unique identifier for the clip
            voice_name: Optional voice name for Riva TTS
            presenter_image: Optional custom presenter image URL
            
        Returns:
            Dict with audio_url, video_url, thumbnail_url, and status
        """
        try:
            logger.info(f"Starting video generation for clip {clip_id}")
            
            # 1. Import Riva service
            from app.services.riva_service import RivaService
            riva_service = RivaService()
            
            # 2. Generate audio with Riva TTS
            logger.info(f"Generating audio for clip {clip_id}")
            audio_data = await riva_service.synthesize_speech(
                text=script,
                voice_name=voice_name
            )
            
            # 3. Upload audio to GCS
            audio_url = await self._upload_audio(clip_id, audio_data)
            logger.info(f"Audio uploaded to {audio_url}")
            
            # 4. Generate video with D-ID
            logger.info(f"Generating video for clip {clip_id}")
            video_url = await self._generate_video_with_did(
                script=script, 
                audio_url=audio_url,
                clip_id=clip_id,
                presenter_image=presenter_image
            )
            
            # 5. Generate thumbnail
            thumbnail_url = await self._generate_thumbnail(clip_id, video_url)
            
            # 6. Calculate video duration from audio
            duration = await self._calculate_duration(audio_data)
            
            logger.info(f"Video generation completed for clip {clip_id}")
            
            return {
                "clip_id": clip_id,
                "audio_url": audio_url,
                "video_url": video_url,
                "thumbnail_url": thumbnail_url,
                "duration": duration,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error generating video for clip {clip_id}: {str(e)}", exc_info=True)
            raise
    
    async def _upload_audio(self, clip_id: str, audio_data: bytes) -> str:
        """Upload audio file to Google Cloud Storage"""
        blob_name = f"clips/{clip_id}/audio.wav"
        blob = self.video_bucket.blob(blob_name)
        
        # Upload with metadata
        blob.metadata = {
            "clip_id": clip_id,
            "created_at": datetime.utcnow().isoformat(),
            "content_type": "audio/wav"
        }
        
        blob.upload_from_string(audio_data, content_type="audio/wav")
        blob.make_public()
        
        logger.info(f"Audio uploaded: {blob.public_url}")
        return blob.public_url
    
    async def _generate_video_with_did(
        self, 
        script: str, 
        audio_url: str,
        clip_id: str,
        presenter_image: Optional[str] = None
    ) -> str:
        """
        Generate video using D-ID API
        
        Args:
            script: Text script (for subtitle generation if needed)
            audio_url: Public URL of the audio file
            clip_id: Clip identifier
            presenter_image: Custom presenter image URL
        """
        try:
            # Default presenter image if not provided
            if not presenter_image:
                presenter_image = "https://create-images-results.d-id.com/default-presenter-image.jpg"
            
            async with httpx.AsyncClient(timeout=300.0) as client:
                # Create talk video
                create_payload = {
                    "script": {
                        "type": "audio",
                        "audio_url": audio_url
                    },
                    "source_url": presenter_image,
                    "config": {
                        "fluent": True,
                        "pad_audio": 0.0,
                        "stitch": True
                    }
                }
                
                logger.info(f"Creating D-ID talk for clip {clip_id}")
                response = await client.post(
                    f"{self.did_url}/talks",
                    headers={
                        "Authorization": f"Basic {self.did_api_key}",
                        "Content-Type": "application/json"
                    },
                    json=create_payload
                )
                response.raise_for_status()
                talk_data = response.json()
                talk_id = talk_data["id"]
                
                logger.info(f"D-ID talk created with ID: {talk_id}")
                
                # Poll for completion
                video_url = await self._poll_did_video(client, talk_id)
                
                # Download and upload to GCS
                final_url = await self._download_and_upload_video(
                    client, 
                    video_url, 
                    clip_id
                )
                
                return final_url
                
        except httpx.HTTPStatusError as e:
            logger.error(f"D-ID API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error generating video with D-ID: {str(e)}", exc_info=True)
            raise
    
    async def _poll_did_video(
        self, 
        client: httpx.AsyncClient, 
        talk_id: str,
        max_attempts: int = 90,
        poll_interval: int = 2
    ) -> str:
        """
        Poll D-ID for video completion with exponential backoff
        
        Args:
            client: HTTP client
            talk_id: D-ID talk ID
            max_attempts: Maximum polling attempts
            poll_interval: Initial interval between polls (seconds)
        """
        for attempt in range(max_attempts):
            try:
                response = await client.get(
                    f"{self.did_url}/talks/{talk_id}",
                    headers={"Authorization": f"Basic {self.did_api_key}"}
                )
                response.raise_for_status()
                data = response.json()
                
                status = data.get("status")
                logger.info(f"D-ID talk {talk_id} status: {status} (attempt {attempt + 1}/{max_attempts})")
                
                if status == "done":
                    return data["result_url"]
                elif status == "error":
                    error_msg = data.get("error", {})
                    raise Exception(f"D-ID video generation failed: {error_msg}")
                elif status in ["created", "started"]:
                    # Still processing, wait and retry
                    await asyncio.sleep(poll_interval)
                else:
                    logger.warning(f"Unknown D-ID status: {status}")
                    await asyncio.sleep(poll_interval)
                    
            except httpx.HTTPStatusError as e:
                logger.error(f"Error polling D-ID: {e.response.status_code}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(poll_interval)
                else:
                    raise
        
        raise Exception(f"Video generation timed out after {max_attempts} attempts")
    
    async def _download_and_upload_video(
        self, 
        client: httpx.AsyncClient, 
        video_url: str, 
        clip_id: str
    ) -> str:
        """Download video from D-ID and upload to Google Cloud Storage"""
        logger.info(f"Downloading video from D-ID: {video_url}")
        
        # Download video
        response = await client.get(video_url)
        response.raise_for_status()
        video_data = response.content
        
        logger.info(f"Video downloaded, size: {len(video_data)} bytes")
        
        # Upload to GCS
        blob_name = f"clips/{clip_id}/video.mp4"
        blob = self.video_bucket.blob(blob_name)
        
        # Set metadata
        blob.metadata = {
            "clip_id": clip_id,
            "created_at": datetime.utcnow().isoformat(),
            "source": "d-id"
        }
        
        blob.upload_from_string(video_data, content_type="video/mp4")
        blob.make_public()
        
        logger.info(f"Video uploaded to GCS: {blob.public_url}")
        return blob.public_url
    
    async def _generate_thumbnail(self, clip_id: str, video_url: str) -> str:
        """
        Generate thumbnail from video using ffmpeg
        Falls back to placeholder if ffmpeg not available
        """
        try:
            import subprocess
            import tempfile
            
            # Download video to temp file
            async with httpx.AsyncClient() as client:
                response = await client.get(video_url)
                response.raise_for_status()
                
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as video_file:
                    video_file.write(response.content)
                    video_path = video_file.name
                
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as thumb_file:
                    thumb_path = thumb_file.name
                
                # Generate thumbnail at 1 second mark
                subprocess.run([
                    'ffmpeg', '-i', video_path,
                    '-ss', '00:00:01',
                    '-vframes', '1',
                    '-vf', 'scale=640:360',
                    thumb_path
                ], check=True, capture_output=True)
                
                # Upload thumbnail
                with open(thumb_path, 'rb') as f:
                    thumbnail_data = f.read()
                
                blob_name = f"clips/{clip_id}/thumbnail.jpg"
                blob = self.video_bucket.blob(blob_name)
                blob.upload_from_string(thumbnail_data, content_type="image/jpeg")
                blob.make_public()
                
                # Cleanup
                os.unlink(video_path)
                os.unlink(thumb_path)
                
                return blob.public_url
                
        except Exception as e:
            logger.warning(f"Could not generate thumbnail: {str(e)}, using placeholder")
            # Return placeholder thumbnail
            return f"https://via.placeholder.com/640x360/007AFF/FFFFFF?text=Motivational+Clip"
    
    async def _calculate_duration(self, audio_data: bytes) -> int:
        """
        Calculate audio/video duration in seconds
        Uses wave library to read WAV file metadata
        """
        try:
            import wave
            import io
            
            with wave.open(io.BytesIO(audio_data), 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration = frames / float(rate)
                return int(duration)
        except Exception as e:
            logger.warning(f"Could not calculate duration: {str(e)}, returning default")
            return 60  # Default 60 seconds
    
    async def batch_generate_clips(
        self, 
        clips_data: List[Dict],
        job_id: str
    ) -> List[Dict]:
        """
        Generate multiple clips in batch
        
        Args:
            clips_data: List of dicts with 'script', 'clip_id', 'voice_name'
            job_id: Processing job ID
            
        Returns:
            List of generation results
        """
        results = []
        
        for i, clip_data in enumerate(clips_data):
            try:
                logger.info(f"Generating clip {i+1}/{len(clips_data)} for job {job_id}")
                
                result = await self.generate_video(
                    script=clip_data['script'],
                    clip_id=clip_data['clip_id'],
                    voice_name=clip_data.get('voice_name'),
                    presenter_image=clip_data.get('presenter_image')
                )
                
                results.append({
                    "success": True,
                    "clip_id": clip_data['clip_id'],
                    "result": result
                })
                
            except Exception as e:
                logger.error(f"Failed to generate clip {clip_data['clip_id']}: {str(e)}")
                results.append({
                    "success": False,
                    "clip_id": clip_data['clip_id'],
                    "error": str(e)
                })
        
        return results
    
    async def publish_generation_task(self, clip_data: Dict) -> str:
        """
        Publish video generation task to Pub/Sub for async processing
        
        Args:
            clip_data: Clip generation data
            
        Returns:
            Message ID
        """
        topic_path = self.publisher.topic_path(
            settings.GCP_PROJECT_ID,
            settings.PUBSUB_TOPIC_VIDEO_PROCESSING
        )
        
        message_data = json.dumps(clip_data).encode('utf-8')
        future = self.publisher.publish(topic_path, message_data)
        message_id = future.result()
        
        logger.info(f"Published video generation task: {message_id}")
        return message_id
