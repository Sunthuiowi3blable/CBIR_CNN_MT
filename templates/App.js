import React, { useState } from 'react';
import {
  Upload,
  SearchIcon,
  Image as ImageIcon
} from 'lucide-react';

export default function ImageSearchApp() {
  const [previewImage, setPreviewImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [searchResults, setSearchResults] = useState([]);
  const [isDragging, setIsDragging] = useState(false);

  const handleImageSelect = (file) => {
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => setPreviewImage(e.target.result);
      reader.readAsDataURL(file);
    }
  };

  const handleSearch = async (file) => {
    if (!file) return;

    setIsLoading(true);
    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await fetch('/search', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();

      if (data.error) throw new Error(data.error);
      setSearchResults(data.results);
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-pink-50 p-4 md:p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header Section */}
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent mb-4">
            Intelligent Image Search
          </h1>
          <p className="text-gray-600 text-lg">
            Discover visually similar images using advanced AI technology
          </p>
        </div>

        {/* Upload Section */}
        <div className="bg-white rounded-3xl shadow-xl p-8 mb-12">
          <div
            className={`
              border-4 border-dashed rounded-2xl p-8 text-center
              transition-all duration-300 ease-in-out
              ${isDragging ? 'border-pink-400 bg-pink-50' : 'border-purple-200 hover:border-purple-400'}
              ${previewImage ? 'border-opacity-50' : ''}
            `}
            onDragEnter={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={(e) => {
              e.preventDefault();
              setIsDragging(false);
            }}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              setIsDragging(false);
              handleImageSelect(e.dataTransfer.files[0]);
            }}
          >
            {!previewImage ? (
              <div className="space-y-4">
                <div className="w-20 h-20 mx-auto bg-gradient-to-br from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
                  <Upload className="w-10 h-10 text-white" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-gray-800">
                    Drop your image here
                  </h3>
                  <p className="text-gray-500 mt-2">
                    or click to browse
                  </p>
                </div>
              </div>
            ) : (
              <img
                src={previewImage}
                alt="Preview"
                className="max-h-64 mx-auto rounded-lg shadow-lg"
              />
            )}
          </div>

          {/* Search Button */}
          {previewImage && (
            <div className="mt-8 text-center">
              <button
                className={`
                  px-8 py-4 rounded-full font-semibold text-white
                  transition-all duration-300 ease-in-out
                  bg-gradient-to-r from-purple-600 to-pink-600
                  hover:from-purple-700 hover:to-pink-700
                  focus:ring-4 focus:ring-purple-300
                  flex items-center justify-center space-x-2
                  mx-auto
                  ${isLoading ? 'opacity-75 cursor-not-allowed' : ''}
                `}
                disabled={isLoading}
              >
                {isLoading ? (
                  <div className="w-6 h-6 border-4 border-white border-t-transparent rounded-full animate-spin" />
                ) : (
                  <>
                    <SearchIcon className="w-5 h-5" />
                    <span>Search Similar Images</span>
                  </>
                )}
              </button>
            </div>
          )}
        </div>

        {/* Results Grid */}
        {searchResults.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {searchResults.map((result, index) => (
              <div
                key={index}
                className="bg-white rounded-2xl overflow-hidden shadow-lg transform transition-all duration-300 hover:-translate-y-2 hover:shadow-xl"
              >
                <div className="relative aspect-w-16 aspect-h-12">
                  <img
                    src={`data:image/jpeg;base64,${result.image}`}
                    alt={`Similar image ${index + 1}`}
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <span className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-full text-sm font-semibold">
                      {(result.score * 100).toFixed(1)}% Match
                    </span>
                  </div>
                  <p className="text-gray-600 text-sm truncate">
                    {result.path.split('/').pop()}
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}