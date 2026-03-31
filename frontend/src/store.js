// Import the reactive and ref functions from Vue
import { reactive } from 'vue';

const buildApiUrl = () => {
    const basePath = (import.meta.env.BASE_URL || '/').replace(/\/$/, '');
    return `${basePath}/backend`;
};

// Define a reactive store to manage global state across the app
export const store = reactive({
    videoUrls: [], // Stores the URLs of the fetched unblurred photos
    apiUrl: buildApiUrl(), // Base URL for API requests proxied by Nginx/reverse proxy
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
