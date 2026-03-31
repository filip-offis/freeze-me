// Import the reactive and ref functions from Vue
import { reactive } from 'vue';

// Define a reactive store to manage global state across the app
export const store = reactive({
    videoUrls: [], // Stores the URLs of the fetched unblurred photos
    apiUrl: '', // Base URL for API requests (empty = relative, proxied by Nginx in Docker)
    selectedVideo: null, // Stores the selected image
    selectedVideoId: null,
    segmentedFrame: null,
    cutVideoFile: null,
    selectedBackground: null,
    totalFrames: 0,
    steps: {
        videoEditing: false,
        segmentation: false,
        mainEffect: false,
        afterEffect: false
    }
});