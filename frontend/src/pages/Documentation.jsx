import React from "react";
import { BookOpen, Video, FileText, Download } from "lucide-react";

const Documentation = () => {
  return (
    <div className="w-full max-w-6xl mx-auto space-y-8 animate-fade-in">
      {/* Header Section */}
      <div className="glass-panel p-8 rounded-2xl border-l-4 border-l-primary relative overflow-hidden">
        <div className="absolute top-0 right-0 p-8 opacity-10">
          <BookOpen className="w-32 h-32 text-indigo-500" />
        </div>
        <div className="relative z-10">
          <h1 className="text-3xl font-bold text-slate-800 mb-4 flex items-center gap-3">
            <BookOpen className="w-8 h-8 text-primary" />
            <p className="text-black-400">Benchmark Note:</p>
          </h1>
          <p className="text-red-400 text-lg leading-relaxed max-w-3xl">
            This application utilizes a multi-agent pipeline involving ~1.4GB of
            model weights. While local inference on an NVIDIA GPU achieves a
            runtime of 4-5 minutes, the current Hugging Face Space runs on 2
            vCPUs (16GB RAM) without GPU acceleration. Consequently, users
            should expect a processing duration of 20-30 minutes for the full
            pipeline.
          </p>
        </div>
      </div>

      {/* Video Section */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Video Card 1 */}
        <div className="glass-panel p-6 rounded-2xl flex flex-col hover:shadow-indigo-500/10 transition-all duration-300 group">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2.5 bg-indigo-50 rounded-lg group-hover:bg-indigo-100 transition-colors">
              <Video className="w-6 h-6 text-primary" />
            </div>
            <h2 className="text-xl font-bold text-slate-700">
              System Overview
            </h2>
          </div>
          <div className="bg-slate-900 rounded-xl overflow-hidden shadow-lg aspect-video mb-4 relative group-hover:scale-[1.01] transition-transform duration-300">
            <video
              className="w-full h-full object-cover"
              controls
              preload="metadata"
            >
              <source
                src={`${import.meta.env.BASE_URL}VID_20260201_013152_web.mp4`}
                type="video/mp4"
              />
              Your browser does not support the video tag.
            </video>

          </div>
          <p className="text-slate-500 text-sm">
            A comprehensive overview of the railway inspection system,
            demonstrating key features and workflow capability.
          </p>
        </div>

        {/* Video Card 2 */}
        <div className="glass-panel p-6 rounded-2xl flex flex-col hover:shadow-indigo-500/10 transition-all duration-300 group">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2.5 bg-indigo-50 rounded-lg group-hover:bg-indigo-100 transition-colors">
              <Video className="w-6 h-6 text-primary" />
            </div>
            <h2 className="text-xl font-bold text-slate-700">Test Video</h2>
          </div>
          <div className="bg-slate-900 rounded-xl overflow-hidden shadow-lg aspect-video mb-4 relative group-hover:scale-[1.01] transition-transform duration-300">
            <video
              className="w-full h-full object-cover"
              controls
              preload="metadata"
            >
              <source
                src={`${import.meta.env.BASE_URL}test4_web.mp4`}
                type="video/mp4"
              />
              Your browser does not support the video tag.
            </video>
          </div>
          <p className="text-slate-500 text-sm">
            This is a sample railway inspection video used to test the system's
            performance and accuracy.
          </p>
          <div className="mt-4">
            <a
              href={`${import.meta.env.BASE_URL}test4_web.mp4`}
              type="video/mp4"
              download
              className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg"
            >
              <Download className="w-4 h-4" />
              Download Test Video
            </a>

          </div>
        </div>
      </div>

      {/* Additional Note/Footer */}
      <div className="glass-panel p-6 rounded-xl flex items-start gap-4 bg-indigo-50/50 border border-indigo-100">
        <FileText className="w-6 h-6 text-indigo-500 flex-shrink-0 mt-1" />
        <div>
          <h3 className="font-semibold text-indigo-900">Note to Users</h3>
          <p className="text-indigo-700/80 text-sm mt-1">
            Ensure you have a stable internet connection for high-quality video
            playback. If you encounter any issues, please check the system
            requirements or contact support.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Documentation;
