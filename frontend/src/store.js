// Import the reactive and ref functions from Vue
import { reactive } from 'vue';

const buildApiUrl = () => {
    const configuredApiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
    return configuredApiUrl.replace(/\/$/, '');
};

// Define a reactive store to manage global state across the app
export const store = reactive({
    videoUrls: [], // Stores the URLs of the fetched unblurred photos
    apiUrl: buildApiUrl(), // Base URL for the local backend API
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
